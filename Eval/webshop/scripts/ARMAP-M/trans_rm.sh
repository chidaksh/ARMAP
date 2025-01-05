rm_url=RM_URL
rm_port=RM_PORT
ssh -L 0.0.0.0:${rm_port}:${rm_url}:${rm_port}  REMOTE_SERVER_ADDRESS