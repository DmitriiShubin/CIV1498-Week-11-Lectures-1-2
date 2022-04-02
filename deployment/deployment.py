import os


def main():

    # get sudo password
    print('Please, enter sudo password')
    sudoPassword = os.environ.get("SUDO_PASS")

    # remove previous deployment
    command = 'docker rmi -f fraud_detection'
    os.system("echo '%s'|sudo -S %s" % (sudoPassword, command))

    # build the image
    command = 'sudo docker build -t fraud_detection .'
    os.system("echo '%s'|sudo -S %s" % (sudoPassword, command))

    # tag the image
    command = 'docker tag fraud_detection:latest 919713240746.dkr.ecr.ca-central-1.amazonaws.com/fraud_detection:latest'
    os.system("echo '%s'|sudo -S %s" % (sudoPassword, command))

    # get ECR credentials
    command = 'aws ecr get-login-password --region ca-central-1 | sudo docker login --username AWS --password-stdin 919713240746.dkr.ecr.ca-central-1.amazonaws.com'
    os.system(command)

    # push the image
    command = 'docker push 919713240746.dkr.ecr.ca-central-1.amazonaws.com/fraud_detection:latest'
    os.system("echo '%s'|sudo -S %s" % (sudoPassword, command))

    # # deploy using serverless
    command = 'serverless deploy --stage dev'
    os.system(command)


if __name__ == '__main__':
    main()
