import hashlib
from streamlit_float import *


def generate_course_id(file):
    file_hash = hashlib.md5(file).hexdigest()
    return file_hash