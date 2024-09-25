
create database DS2023;
use DS2023;#사용할 db 결정

create user 'datascience'@'%' identified by '0000'; #root 권한으로 DS2023라는 데이터베이스에서 식별가능한 user 생성(password는 '0000')
grant all privileges on DS2023.* to 'datascience'@'%';# 'database' user에게 'university' 데이터베이스에 대한 모든 권한을 부여
#GRANT DROP ON DS2023.* TO 'datascienscorescorece'@'%';
#grant CREATE ON *.* TO 'datascience'@'%';# 'database' user에게 모든 데이터베이스에 대한 CREATE 권한을 부여
flush privileges;
#MySQL 서버에 대한 권한 캐시를 재로드하여 새로운 권한을 적용하는 데 사용. 
#실행하면 현재 메모리에 있는 권한 정보가 다시 로드되어 적용
SHOW GRANTS FOR 'datascience'@'%';
