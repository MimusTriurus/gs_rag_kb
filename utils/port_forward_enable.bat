@echo off

REM Порт, который будет слушать Windows извне (например, 8000)
set listen_port=8000

REM Порт, к которому будет подключаться Windows (порт, на котором реально слушает Uvicorn, 5000)
set connect_port=5000

REM Адрес, на котором Windows будет слушать (0.0.0.0 - все сетевые интерфейсы)
set listen_address=0.0.0.0

REM Адрес, к которому будет подключаться Windows (127.0.0.1 - localhost, где Uvicorn)
set connect_address_ipv4=127.0.0.1
set connect_address_ipv6=::1

echo #==============================================#
echo Setting up port forwarding: %listen_port% -> %connect_port%
echo #==============================================#

REM Правило проброса порта IPv4 к IPv4
REM Весь трафик на listen_port (%listen_port%) будет перенаправлен на connect_port (%connect_port%) на localhost (%connect_address_ipv4%).
netsh interface portproxy add v4tov4 listenport=%listen_port% listenaddress=%listen_address% connectport=%connect_port% connectaddress=%connect_address_ipv4%

REM Правило проброса порта IPv4 к IPv6 (если ваш сервер слушает на IPv6, что менее распространено для Uvicorn по умолчанию)
netsh interface portproxy add v4tov6 listenaddress=%listen_address% listenport=%listen_port% connectaddress=%connect_address_ipv6% connectport=%connect_port% protocol=tcp

REM Правило брандмауэра для разрешения входящих соединений на listen_port (%listen_port%)
REM Это позволит внешним устройствам подключаться к listen_port.
netsh advfirewall firewall add rule name="Uvicorn_Proxy_%listen_port%_to_%connect_port%" dir=in action=allow protocol=TCP localport=%listen_port%

echo #----------------------------------------------#
echo.

echo Current Port Proxy rules:
netsh interface portproxy dump

@pause