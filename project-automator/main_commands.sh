function create() {
    cd
    python3 create.py $1
    cd /Documents/Projects/AutomatedProjects/$1
    git init
    git remote add origin git@github.com:jdewgun/$1.git
    touch README.md
    git add .
    git commit -m 'Initial Commit'
    git push -u origin master
    code .
}
