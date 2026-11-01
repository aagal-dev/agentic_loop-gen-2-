from core.agentic_loop import agentic_loop

def main() -> None:
    loop = agentic_loop()
    try:
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
    finally:
        memory_result = loop.end_session()
        if memory_result is not None:
            for message in (memory_result.error, *memory_result.indexing_errors):
                if message:
                    print(f"Memory warning: {message}")


if __name__ == "__main__":
    main()
