import paramiko
import os
import sys

# Server configuration
HOST = "task2vec.com"
USERNAME = "adloccbvmx"
PASSWORD = "Euroopp4-2025"
REMOTE_BASE = "/home/adloccbvmx/domains/task2vec.com/public_html/"

# Files to upload: (local_path, remote_path)
FILES = [
    (
        r"C:\Users\jussi\ai-driving-license\post3_carousel.html",
        REMOTE_BASE + "post3_carousel.html"
    ),
    (
        r"C:\Users\jussi\ai-driving-license\post4_carousel.html",
        REMOTE_BASE + "post4_carousel.html"
    ),
    (
        r"C:\Users\jussi\ai-driving-license\linkedin_posts.html",
        REMOTE_BASE + "linkedin_posts.html"
    ),
]

def deploy():
    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print(f"Connecting to {HOST} as {USERNAME}...")
    try:
        ssh.connect(hostname=HOST, username=USERNAME, password=PASSWORD, timeout=30)
    except Exception as e:
        print(f"ERROR connecting: {e}")
        sys.exit(1)

    print("Connected. Opening SFTP session...")
    sftp = ssh.open_sftp()

    results = []
    for local_path, remote_path in FILES:
        filename = os.path.basename(local_path)
        print(f"\nUploading: {filename}")
        print(f"  Local : {local_path}")
        print(f"  Remote: {remote_path}")

        if not os.path.exists(local_path):
            print(f"  ERROR: Local file not found!")
            results.append((filename, False, "Local file not found"))
            continue

        local_size = os.path.getsize(local_path)
        print(f"  Local size: {local_size} bytes")

        try:
            sftp.put(local_path, remote_path)
            # Verify by checking remote file size
            remote_stat = sftp.stat(remote_path)
            remote_size = remote_stat.st_size
            print(f"  Remote size: {remote_size} bytes")
            if local_size == remote_size:
                print(f"  SUCCESS: File uploaded and verified (sizes match).")
                results.append((filename, True, f"{remote_size} bytes"))
            else:
                print(f"  WARNING: Size mismatch! local={local_size}, remote={remote_size}")
                results.append((filename, False, f"Size mismatch: local={local_size} remote={remote_size}"))
        except Exception as e:
            print(f"  ERROR uploading {filename}: {e}")
            results.append((filename, False, str(e)))

    sftp.close()
    ssh.close()

    print("\n" + "="*60)
    print("DEPLOYMENT SUMMARY")
    print("="*60)
    all_ok = True
    for filename, success, detail in results:
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {filename} — {detail}")
        if not success:
            all_ok = False
    print("="*60)
    if all_ok:
        print("All files deployed successfully.")
    else:
        print("One or more files failed to deploy.")
        sys.exit(1)

if __name__ == "__main__":
    deploy()
