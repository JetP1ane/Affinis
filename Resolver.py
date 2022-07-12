import socket

class Resolver():

    def resolveHost(self, subdomain):
        try:
            check = socket.gethostbyname(subdomain)
            if check:
                return ("[LSTM][HostExists] " + subdomain  + " => " + check)
        except Exception as err:
            exit

