/* Shared clipboard support for the browser and Windows WebView2. */
window.ShaperClipboard = {
  async copy(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try {
        await navigator.clipboard.writeText(text);
        return;
      } catch (_) { /* Try the local WebView/legacy fallback. */ }
    }
    const previousFocus = document.activeElement;
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0";
    document.body.appendChild(textarea);
    try {
      textarea.select();
      textarea.setSelectionRange(0, text.length);
      if (!document.execCommand("copy")) throw new Error("无法访问剪贴板，请下载 Lua 文件");
    } finally {
      textarea.remove();
      if (previousFocus && previousFocus.focus) previousFocus.focus();
    }
  },
};
