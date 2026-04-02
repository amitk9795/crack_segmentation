# crack_segmentation

## Streamlit Cloud deployment notes

- Keep `packages.txt` minimal.
- Use only:

```text
libgl1
```

- Do not add `libglib2.0-0` on Streamlit Cloud. It can trigger Debian dependency conflicts (`libffi7`/`libpcre3`) and block app startup.

If deployment fails after dependency edits, restart the app and clear cache in Streamlit Cloud.