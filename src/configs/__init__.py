from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=["src/configs/config.yaml"], lowercase_read=True, environments=True
)
