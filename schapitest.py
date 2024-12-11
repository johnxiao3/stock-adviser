from access_key import *

from time import sleep
import schwabdev
import datetime
import logging
import os


def main():

    # set logging level
    #logging.basicConfig(level=logging.INFO)

    # create client
    client = schwabdev.Client(app_key, app_secret, callback_url)
    print(client)

    
    print("\nGet account number and hashes for linked accounts")
    linked_accounts = client.account_linked().json()
    print(linked_accounts)
    account_hash = linked_accounts[0].get('hashValue') # this will get the first linked account
    print(account_hash)
    sleep(3)
    '''
    print("\nGet details for all linked accounts")
    print(client.account_details_all().json())
    sleep(3)

    print("\nGet specific account positions (uses default account, can be changed)")
    print(client.account_details(account_hash, fields="positions").json())
    sleep(3)

    # get orders for a linked account
    print("\nGet orders for a linked account")
    print(client.account_orders(account_hash, datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30), datetime.datetime.now(datetime.timezone.utc)).json())
    sleep(3)
    '''

    # Define variables for symbol and quantity
    symbol = "OPK"
    quantity = 1

    # Get the quote
    ret = client.quote(symbol).json()

    # Get the extended ask price from the response
    ask_price = str(ret[symbol]['quote']['askPrice'])

    # Create the order using the dynamic price and variables
    order = {
        "orderType": "LIMIT",
        "session": "NORMAL",
        "duration": "DAY",
        "orderStrategyType": "SINGLE",
        "price": ask_price,
        "orderLegCollection": [
            {
                "instruction": "BUY",
                "quantity": quantity,
                "instrument": {
                    "symbol": symbol,
                    "assetType": "EQUITY"
                }
            }
        ]
    }

    resp = client.order_place(account_hash, order)
    print("\nPlace an order:")
    print(f"Response code: {resp}")
    # get the order ID - if order is immediately filled then the id might not be returned
    order_id = resp.headers.get('location', '/').split('/')[-1]
    print(f"Order id: {order_id}")
    sleep(3)



if __name__ == '__main__':
    print("Welcome to The Unofficial Schwab Python Wrapper!")
    print("Github: https://github.com/tylerebowers/Schwab-API-Python")
    print("API documentation: https://github.com/tylerebowers/Schwab-API-Python/blob/master/docs/api.md")
    print("Client documentation: https://github.com/tylerebowers/Schwab-API-Python/blob/master/docs/client.md")
    main()  # call the user code above