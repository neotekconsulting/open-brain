import sys, json
from pathlib import Path

path = Path(r'C:/Users/jeffj/open-brain/mcp-brain/server.py')
code = path.read_text(encoding='utf-8')
compile(code, str(path), 'exec')
print('PY_COMPILE_OK')

# Minimal AST-style checks instead of importing deps
assert 'async def semantic_search' in code, 'semantic_search missing'
assert 'async def list_recent' in code, 'list_recent missing'
assert 'async def stats' in code, 'stats missing'
assert 'async def capture_thought' in code, 'capture_thought missing'
assert 'app = FastMCP(' in code, 'FastMCP app missing'
assert 'async with asyncpg.create_pool(' in code, 'pg pool missing'
assert 'async with httpx.AsyncClient(' in code, 'httpx client missing'
assert 'transport="stdio"' in code and 'transport="http"' in code, 'transports missing'
assert 'write_claude_desktop_config' in code, 'claude writer missing'
assert 'write_cursor_mcp_config' in code, 'cursor writer missing'
assert 'fastmcp==3.4.2' in Path(r'C:/Users/jeffj/open-brain/mcp-brain/requirements.txt').read_text(), 'requirements mismatch'
print('CHECKS_PASS')
