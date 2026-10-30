from core.agentic_loop import agentic_loop

def main() -> None:
    loop = agentic_loop()
    while True:
        try:
            user = input(">>> ")
        except EOFError:
            break

        if user.strip().lower() == "q":
            break

        result = loop.run(user)
        if result.status == "success":
            print(result.decision)
        else:
            print(f"Error: {result.error}")


if __name__ == "__main__":
    main()
