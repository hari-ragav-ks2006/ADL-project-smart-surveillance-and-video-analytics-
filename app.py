import sys
import os
import subprocess

def is_running_under_streamlit():
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

def main():
    dashboard_app = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")

    # If run via `python app.py` directly (outside `streamlit run`), invoke streamlit CLI automatically!
    if not is_running_under_streamlit():
        print(f"🚀 Launching Streamlit dashboard on dashboard/app.py...")
        cmd = [sys.executable, "-m", "streamlit", "run", dashboard_app] + sys.argv[1:]
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nDashboard closed.")
        sys.exit(0)
    else:
        # Running under Streamlit context
        with open(dashboard_app, "r", encoding="utf-8") as f:
            code = compile(f.read(), dashboard_app, 'exec')
            exec(code, {'__name__': '__main__', '__file__': dashboard_app})

if __name__ == "__main__":
    main()
