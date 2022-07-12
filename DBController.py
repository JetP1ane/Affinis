import sqlite3


class DBController():

    def __init__(self, domain):
        self.domain = domain
        self.conn = sqlite3.connect("./data/osiris.db")

    def main(self):
        print("[+] Starting DB Controller..")

    def selectSubs(self):

        returnData = []
        
        cursor = self.conn.cursor()

        cursor.execute('''SELECT * FROM subdomains WHERE subdomain LIKE '%''' + self.domain + '''%' LIMIT 40''')

        result = cursor.fetchall()
        
        for item in result:
            returnData.append(item[0])

        return returnData

        self.conn.close()