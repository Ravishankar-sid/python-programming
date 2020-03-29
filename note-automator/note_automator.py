import sys
import os


class NoteAutomator:
    python = ".py"
    java = ".java"
    dart = ".dart"
    text = ".txt"
    yaml = ".yaml"
    json = ".json"

    extensions = {
        ".py": python,
        ".txt": text,
        ".java": java,
        ".dart": dart,
        ".yaml": yaml,
        ".json": json,
    }

    extension = ""
    folder_name = ""
    file_name = ""
    foundFileExtension = ""
    path = os.getcwd()

    def getArgs(self, ext_num, fold_num):
        try:
            self.extension = str(sys.argv[ext_num]).lower()
            self.extension = self.extensions[self.extension]
        except Exception:
            self.extension = ".txt"
        try:
            self.folder_name = str(sys.argv[fold_num])
        except Exception:
            self.folder_name = "General"
        try:
            self.file_name = str(sys.argv[2])
        except Exception:
            print("Name your note")
            sys.exit()

    def create_note_and_folder(self):
        os.chdir("./Notes")

        self.file_name = self.file_name + self.extension
        if os.path.isdir("./" + self.folder_name):
            os.chdir("./" + self.folder_name)
        else:
            os.mkdir(self.folder_name)
            os.chdir("./" + self.folder_name)

        if not os.path.isfile("./" + self.file_name):
            open(self.file_name, "a").close()

        os.system("subl " + self.file_name)

    def search_file_in_folder(self, folder):
        if os.path.isdir(self.path + "/" + folder):
            self.path = self.path + "/" + folder
            self.find_file(self.file_name, "", self.path)
        else:
            self.path = self.findFolder(folder, "", self.path)
            self.find_file(self.file_name, "", self.path)

        os.system("subl " + self.path)

    def find_file(self, file_to_find, folder_to_search, path):
        file_exists = False
        folder_path = ""
        for subdir, dirs, files in os.walk(path + folder_to_search):
            for dir_ in dirs:
                if dir_.lower() == self.folder_name.lower():
                    folder_path = ""
                    folder_path = subdir + "/" + self.folder_name
            for file_ in files:
                name = ""
                for i in range(len(str(file_))):
                    if len(str(self.file_name)) > i:
                        if str(file_).lower()[i] == str(self.file_name).lower()[i]:
                            name = name + str(file_)[i]
                            if len(name) > len(self.file_name) * 0.8:
                                self.path = os.path.join(subdir, file_)
                                file_exists = True
                                break
        if not file_exists:
            self.path = os.path.join(folder_path, self.file_name + self.extension)
            open(self.path, "a").close()


if __name__ == "__main__":
    notes = NoteAutomator()

    command = str(sys.argv[1])

    if command == "nfe":
        notes.getArgs(4, 3)
        notes.create_note_and_folder()

    if command == "on":
        notes.getArgs(4, 3)
        try:
            notes.search_file_in_folder(str(sys.argv[3]))
        except Exception:
            notes.search_file_in_folder("")
    if command == "ne":
        notes.getArgs(3, 10)
        notes.search_file_in_folder("")
