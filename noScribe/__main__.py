import noScribe

if __name__ == "__main__":
    try:
        noScribe.main.noScribeMain()
    except Exception as e:
        print(e)
        SystemExit(1)
