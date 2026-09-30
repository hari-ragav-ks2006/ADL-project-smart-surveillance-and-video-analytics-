from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import os

import cv2
import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from sqlalchemy import and_, desc
from sqlalchemy.orm import Session

from database import AlertRecord, DatabaseManager, SystemLogRecord, get_utc_now

logger = logging.getLogger("SurveillanceLogger")


class SurveillanceLogger:
    """Handles logging to Database, saving annotated screenshots, and Excel generation."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        screenshots_dir: str = "logs/screenshots",
        excel_path: str = "logs/surveillance_alerts.xlsx",
    ):
        self.db_manager = db_manager
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.excel_path = Path(excel_path)
        self.excel_path.parent.mkdir(parents=True, exist_ok=True)

    def log_alert(
        self,
        alert_type: str,
        severity: str,
        module_name: str,
        message: str,
        tracked_ids: Optional[List[int]] = None,
        zone_name: Optional[str] = None,
        frame: Optional[np.ndarray] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AlertRecord:
        """Create a new alert record, optionally save annotated frame, and commit to DB."""
        now = get_utc_now()
        tracked_ids = tracked_ids or []
        details = details or {}

        screenshot_path_str = None
        if frame is not None and frame.size > 0:
            filename = f"ALERT_{alert_type}_{now.strftime('%Y%m%d_%H%M%S_%f')[:19]}.jpg"
            save_path = self.screenshots_dir / filename
            # Write annotated screenshot
            cv2.imwrite(str(save_path), frame)
            screenshot_path_str = str(save_path).replace("\\", "/")

        with self.db_manager.get_session() as session:
            record = AlertRecord(
                timestamp=now,
                alert_type=alert_type,
                severity=severity.upper(),
                module_name=module_name,
                zone_name=zone_name,
                tracked_ids=json.dumps(tracked_ids),
                message=message,
                resolution_status="ACTIVE",
                screenshot_path=screenshot_path_str,
                details_json=json.dumps(details),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            alert_id = record.id

        # Trigger background excel mirror update
        try:
            self.export_to_excel()
        except Exception as e:
            logger.warning(f"[LOGGER] Exception during export_to_excel: {e}")

        # Re-fetch detached record representation
        with self.db_manager.get_session() as session:
            return session.get(AlertRecord, alert_id)

    def resolve_alert(
        self,
        alert_id: int,
        resolution_status: str = "RESOLVED",
    ) -> Optional[AlertRecord]:
        """Mark an alert as resolved or escalated."""
        with self.db_manager.get_session() as session:
            record = session.get(AlertRecord, alert_id)
            if record:
                record.resolution_status = resolution_status
                record.resolved_at = get_utc_now()
                session.commit()
                session.refresh(record)
                try:
                    self.export_to_excel()
                except Exception as e:
                    logger.warning(f"[LOGGER] Exception during export_to_excel in resolve_alert: {e}")
                return record
        return None

    def log_system_event(self, event_type: str, message: str) -> SystemLogRecord:
        """Log general system events (e.g. startup, simulation dispatches)."""
        now = get_utc_now()
        with self.db_manager.get_session() as session:
            rec = SystemLogRecord(
                timestamp=now,
                event_type=event_type,
                message=message,
            )
            session.add(rec)
            session.commit()
            session.refresh(rec)
            return rec

    def query_alerts(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        status: Optional[str] = None,
        module_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Filter alerts by date range, severity, type, module, and resolution status."""
        with self.db_manager.get_session() as session:
            query = session.query(AlertRecord)
            filters = []

            if start_date:
                filters.append(AlertRecord.timestamp >= start_date)
            if end_date:
                filters.append(AlertRecord.timestamp <= end_date)
            if severity:
                filters.append(AlertRecord.severity == severity.upper())
            if alert_type:
                filters.append(AlertRecord.alert_type == alert_type)
            if status:
                filters.append(AlertRecord.resolution_status == status.upper())
            if module_name:
                filters.append(AlertRecord.module_name.ilike(f"%{module_name}%"))

            if filters:
                query = query.filter(and_(*filters))

            query = query.order_by(desc(AlertRecord.timestamp)).limit(limit)
            records = query.all()
            return [r.to_dict() for r in records]

    def export_to_excel(self) -> str:
        """Mirror current database alerts into a formatted Excel sheet with openpyxl."""
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Surveillance Alerts"

        # Headers
        headers = [
            "ID",
            "Timestamp (UTC)",
            "Alert Type",
            "Severity",
            "Module",
            "Zone",
            "Tracked IDs",
            "Message",
            "Status",
            "Resolved At",
            "Screenshot File",
        ]
        ws.append(headers)

        # Style definitions
        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1F2937", end_color="1F2937", fill_type="solid")
        thin_border = Border(
            left=Side(style="thin", color="D1D5DB"),
            right=Side(style="thin", color="D1D5DB"),
            top=Side(style="thin", color="D1D5DB"),
            bottom=Side(style="thin", color="D1D5DB"),
        )

        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border

        # Severity fills
        sev_fills = {
            "CRITICAL": PatternFill(start_color="FCA5A5", end_color="FCA5A5", fill_type="solid"),
            "HIGH": PatternFill(start_color="FDBA74", end_color="FDBA74", fill_type="solid"),
            "MEDIUM": PatternFill(start_color="FEF08A", end_color="FEF08A", fill_type="solid"),
            "LOW": PatternFill(start_color="E0F2FE", end_color="E0F2FE", fill_type="solid"),
        }

        # Populate rows
        with self.db_manager.get_session() as session:
            alerts = (
                session.query(AlertRecord)
                .order_by(desc(AlertRecord.timestamp))
                .limit(1000)
                .all()
            )

            for row_idx, alert in enumerate(alerts, 2):
                screenshot_filename = (
                    Path(alert.screenshot_path).name if alert.screenshot_path else "None"
                )
                row_data = [
                    alert.id,
                    alert.timestamp.strftime("%Y-%m-%d %H:%M:%S") if alert.timestamp else "",
                    alert.alert_type,
                    alert.severity,
                    alert.module_name,
                    alert.zone_name or "N/A",
                    alert.tracked_ids,
                    alert.message,
                    alert.resolution_status,
                    alert.resolved_at.strftime("%Y-%m-%d %H:%M:%S") if alert.resolved_at else "N/A",
                    screenshot_filename,
                ]
                ws.append(row_data)

                # Format row cells
                sev_fill = sev_fills.get(alert.severity, None)
                for col_idx in range(1, len(row_data) + 1):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    cell.font = Font(name="Segoe UI", size=10)
                    cell.border = thin_border
                    cell.alignment = Alignment(vertical="center")
                    if col_idx == 4 and sev_fill:  # Severity column
                        cell.fill = sev_fill
                        cell.alignment = Alignment(horizontal="center", vertical="center")
                    if col_idx in (1, 9):
                        cell.alignment = Alignment(horizontal="center", vertical="center")

        # Auto-fit column widths
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val = str(cell.value or "")
                max_len = max(max_len, len(val))
            ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

        try:
            wb.save(str(self.excel_path))
        except PermissionError:
            logger.warning(f"[LOGGER] Could not write to {self.excel_path} (file is locked or open in Excel).")
        except Exception as e:
            logger.warning(f"[LOGGER] Error saving Excel export: {e}")

        return str(self.excel_path).replace("\\", "/")
