class SessionManager:
    def __init__(self):
        self.recognized_students = set()

    def is_already_recognized(self, name):
        return name in self.recognized_students

    def add_student(self, name):
        if name:
            self.recognized_students.add(name)

    def clear_session(self):
        self.recognized_students.clear()

    def get_all(self):
        return list(self.recognized_students)

