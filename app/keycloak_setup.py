
from keycloak import KeycloakAdmin, KeycloakOpenID
from config import settings

keycloak_admin = KeycloakAdmin(server_url=settings.keycloak_server_url,
                                username='admin',
                                password='admin',
                                realm_name='OBR',
                                verify=True)

keycloak_openid = KeycloakOpenID(server_url=settings.keycloak_server_url,
                                   realm_name=settings.keycloak_realm,
                                   client_id=settings.keycloak_client_id,
                                client_secret_key=settings.keycloak_client_secret)

