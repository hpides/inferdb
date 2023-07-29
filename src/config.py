from configparser import ConfigParser
import os

def config(section='postgresql'):
    
    current = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(current, 'database.ini')
    
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(path)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, path))

    return db