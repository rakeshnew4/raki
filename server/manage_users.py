"""Manage the web UI logins.

  docker exec -it raki_audio python manage_users.py add <username>      (asks for the password)
  docker exec -i  raki_audio python manage_users.py add <username> < pw.txt   (password from stdin)
  docker exec raki_audio python manage_users.py remove <username>
  docker exec raki_audio python manage_users.py list
"""
import getpass
import sys

import webui


def read_password() -> str:
    if sys.stdin.isatty():
        first, second = getpass.getpass("Password: "), getpass.getpass("Repeat: ")
        if first != second:
            sys.exit("Passwords do not match")
        return first
    return sys.stdin.readline().rstrip("\r\n")


def main() -> None:
    args = sys.argv[1:]
    users = webui.load_users()
    if args[:1] == ["list"]:
        print("\n".join(sorted(users)) or "(no users)")
    elif len(args) == 2 and args[0] == "add":
        password = read_password()
        if len(password) < 8:
            sys.exit("Password must be at least 8 characters")
        users[args[1]] = webui.hash_password(password)
        webui.save_users(users)
        print(f"User '{args[1]}' saved")
    elif len(args) == 2 and args[0] == "remove":
        if users.pop(args[1], None) is None:
            sys.exit(f"No user '{args[1]}'")
        webui.save_users(users)
        print(f"User '{args[1]}' removed")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
