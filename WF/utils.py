# utils.py
from twilio.rest import Client

def send_sms_to_admin(message):
    account_sid = 'your_account_sid'
    auth_token = 'your_auth_token'
    client = Client(account_sid, auth_token)

    client.messages.create(
        body=message,
        from_='+your_twilio_number',  # Your Twilio number
        to='+admin_phone_number'      # Admin's number
    )
