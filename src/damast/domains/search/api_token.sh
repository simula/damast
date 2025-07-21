# Create a .env file defining
# the copernicus user and password
# see https://documentation.dataspace.copernicus.eu/APIs/Token.html
#
# CC_USER=<copernicus-user>
# CC_PASSWORD=<copernicus-password>

source .env

export ACCESS_TOKEN=$(curl -d 'client_id=cdse-public' \
                    -d "username=$CC_USER" \
                    -d "password=$CC_PASSWORD" \
                    -d 'grant_type=password' \
                    'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token' | \
                    python3 -m json.tool | grep "access_token" | awk -F\" '{print $4}')
echo $ACCESS_TOKEN

