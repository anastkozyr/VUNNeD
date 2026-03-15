from flask import Flask, render_template, send_from_directory, redirect, request, jsonify, \
    url_for  # ← добавили request!
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, login_manager
from data import db_session
from data.users import User
from forms.user import RegisterForm, LoginForm


app = Flask(__name__)
db_session.global_init("db/users.db")

@app.route('/')
def index():
    return render_template('index.html')

@login_manager.user_loader
def load_user(user_id):
    db_sess = db_session.create_session()
    try:
        return db_sess.query(User).get(int(user_id))
    finally:
        db_sess.close()

@app.route('/new_dialog')
def new_dialog():
    return render_template('new_dialog.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        db_sess = db_session.create_session()
        # Ищем пользователя по имени, а не по email
        try:
            user = db_sess.query(User).filter(User.name == form.name.data).first()

            if user and user.check_password(form.password.data):
                login_user(user, remember=True)  # remember_me убрали
                return redirect("/index.html")  # Перенаправляем в тренажер

            return render_template('login_user.html',
                                   message="Неправильное имя пользователя или пароль",
                                   form=form)
        finally:
            db_sess.close()

    return render_template('login.html', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect("/")

@app.route('/account')
def account():

    if 'user_id' not in db_session:
        return redirect(url_for('login'))

    user_id = current_user.id

    user_dict = {
        'id': current_user.id,
        'username': current_user.username,
        'email': current_user.email,
        'password': current_user.password,
        'role': current_user.role
    }

    return render_template('account.html', user=user_dict)

if __name__ == '__main__':
    app.run(debug=True)