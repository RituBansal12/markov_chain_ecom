from cleaning_scripts.common import ensure_dirs, configure_logging


def main():
    log = configure_logging("prep_dirs")
    ensure_dirs()
    log.info("Ensured data_clean/ and reports/ directories exist.")


if __name__ == "__main__":
    main()
