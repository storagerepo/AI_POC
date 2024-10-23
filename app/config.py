from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    keycloak_server_url: str
    keycloak_realm: str
    keycloak_client_id: str
    keycloak_client_secret: str
    keycloak_public_key:str
    class Config:
        env_file = ".env"


settings = Settings()
