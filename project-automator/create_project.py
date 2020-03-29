import sys
import os
from github import Github

path = "/Documents/Projects/AutomatedProjects/"

username = "" # GitHub Username
password = "" # GitHub Password

def create():
    folder_name = str(sys.argv[1])
    os.makedirs(path + str(folder_name)
    user = Github(username, password).get_user()
    repository = user.create_repo(folder_name)
    print("Succesfully created repository {}".format(folder_name))

if __name__ == "__main__":
    create()
