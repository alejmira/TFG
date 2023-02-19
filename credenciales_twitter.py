# Credenciales para poder escuchar tweets. De esta manera se mantienen separados
# del programa escuchador

# Estos valores salen de la configuración de la aplicación
CONSUMER_TOKEN = "..."
CONSUMER_SECRET = "..."
    
# Estos valores hay que obtenerlos a mano usando el navegador
# 1) Visitar la URL de verificación y conseguir el PIN
#       auth.get_authorization_url()
# 2) Introducir el PIN en auth para obtener access_token y access_token_secret
#       auth.get_access_token(PIN)
# 3) Extraer access_token y access_token_secret y almacenarlos
#       auth.access_token
#       auth.access_token_secret
ACCESS_TOKEN = '...'
ACCESS_TOKEN_SECRET = '...'
