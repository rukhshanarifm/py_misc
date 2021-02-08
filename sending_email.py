'''
I utilized the following resources to understand this process:
	1. https://stackabuse.com/how-to-send-emails-with-gmail-using-python/
	2. https://www.geeksforgeeks.org/send-mail-attachment-gmail-account-using-python/

'''

import smtplib
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
from email.mime.application import MIMEApplication

def send_email():
	'''
	This function allows you to automate email sending tasks via Gmail.
	'''
    msg = MIMEMultipart()
    msg.preamble = 'ENTER EMAIL TEXT (BODY)'

    msg['Subject'] = 'ENTER EMAIL SUBJECT'
    msg['From'] = "FROM EMAIL"
    msg['To'] = "TO EMAIL"
    
    #Can comment out the following incase there is no file to attach
    file = "FILE TO ATTACH"
    attachment=open(file,'rb')

    #Will need to remove part.add_header IF no file attached
    part = MIMEApplication(attachment.read(),'.txt')
    part.add_header('Content-Disposition',"attachement; filename="+file)
    msg.attach(part)

    # Send the message via our own SMTP server.
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)

    '''
    Since gmail is being used here, you'll need to set up Two Step verification.
    You can do that by:
    1. Logging into your gmail account.
    2. Look for the Google Apps button on the top right hand side of the page and click.
    3. Click on Account
    4. Navigate to the Security tab (4th option on the left-most side)
    5. Scroll down to "Signing into Google"
    6. Click of "App Passwords" and verify that you are using your gmail account
    7. Under "Select App", choose the "Mail" option and under "Select Device"
       ,you can click on "Other" and type in "Python"
    8. Generate the passcode and use it below inplace of "GENERATED PASSCODE"
    '''

    server.login("YOUR EMAIL ADDRESS", "GENERATED PASSCODE")
    server.send_message(msg)
    server.quit()

