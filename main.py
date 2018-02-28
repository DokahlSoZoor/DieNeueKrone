import logging
import traceback
from multiprocessing import Process, Queue
from flask import Flask

import model

app = Flask(__name__)

q = Queue()
p = None

def generate_articles(output_q, amount=5):
    for title, body in model.generate_articles('checkpoints/rnn_train_1519647475-248000000', amount=amount):
        output_q.put((title, body))
        print('article generated')

if __name__ == '__main__':

    p = Process(target=f, args=(q,))
    p.daemon = True
    p.start()
    print(q.get())    # prints "[42, None, 'hello']"
    p.join()


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Welcome to Die Neue Krone!'

@app.route('/sample')
def sample():
    """Generate and return an article"""
    # title, body = next(model.generate_articles('checkpoints/rnn_train_1519647475-248000000', amount=1))
    if q.qsize() < 2:
        global p
        if p == None or not p.is_alive():
            p = Process(target=generate_articles, args=(q,))
            p.daemon = True
            p.start()
        return "try again in a moment"
    else:
        title, body = q.get()
        return """<h1>{}</h1><p>{}</p>""".format(title, body.replace('\n', '<br>'))

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    trace:
    {}
    """.format(e, traceback.format_exc()), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
