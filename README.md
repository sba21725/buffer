# Database credentials
hostname = 'mysql-14de7744-student-06ec.e.aivencloud.com'
port=24884
username = 'avnadmin'
password = 'AVNS_es-sxOgVqlmYavz4rjQ'
database_name = 'defaultdb'
ca_cert_path = 'ca.pem'

# Establish a secure SSL connection to the MySQL database
connection = pymysql.connect(
    host=hostname,
    port=port,
    user=username,
    password=password,
    database=database_name,
    ssl={'ca': ca_cert_path}  # SSL configuration
)
