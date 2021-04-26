from colorama import Fore, Style


def info(msg):
    print(Fore.GREEN, end="")
    print(msg, end="")
    print(Style.RESET_ALL)
