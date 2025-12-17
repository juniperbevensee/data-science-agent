import os
import config

# Resolve workspace to absolute path once
WORKSPACE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), config.WORKSPACE_DIR)
)


class SandboxError(Exception):
    """Raised when a path escapes the sandbox."""
    pass


def resolve_path(path: str) -> str:
    """
    Resolve a path safely within the sandbox.
    
    Args:
        path: A relative path (e.g., "data.csv" or "subdir/file.json")
    
    Returns:
        Absolute path within the workspace
    
    Raises:
        SandboxError: If the path would escape the sandbox
    """
    # Reject absolute paths
    if os.path.isabs(path):
        raise SandboxError(f"Absolute paths not allowed: {path}")
    
    # Resolve the full path
    full_path = os.path.abspath(os.path.join(WORKSPACE_PATH, path))
    
    # Ensure it's still within workspace
    if not full_path.startswith(WORKSPACE_PATH + os.sep) and full_path != WORKSPACE_PATH:
        raise SandboxError(f"Path escapes sandbox: {path}")
    
    return full_path


def list_files(subdir: str = "") -> list[str]:
    """List files in the workspace or a subdirectory."""
    target = resolve_path(subdir) if subdir else WORKSPACE_PATH
    
    if not os.path.isdir(target):
        raise SandboxError(f"Not a directory: {subdir}")
    
    return os.listdir(target)


# Test
if __name__ == "__main__":
    print(f"Workspace: {WORKSPACE_PATH}")
    
    # Should work
    print(f"Valid path: {resolve_path('test.csv')}")
    print(f"Valid subdir: {resolve_path('subdir/file.csv')}")
    
    # Should fail
    for bad in ["../etc/passwd", "/etc/passwd", "subdir/../../etc/passwd"]:
        try:
            resolve_path(bad)
            print(f"FAIL: {bad} should have been rejected")
        except SandboxError as e:
            print(f"Blocked: {e}")

