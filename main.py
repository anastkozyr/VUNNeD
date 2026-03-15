from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from functools import wraps

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'VUNNeD'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    role_id = db.Column(db.Integer, db.ForeignKey('user_role.id'), nullable=False)
    password = db.Column(db.String(120), nullable=False)


class UserRole(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    role_name = db.Column(db.SmallInteger, nullable=False)


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        role = UserRole.query.get(user.role_id)
        if not role or role.role_name != 'admin':
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/dialog')
def dialog():
    return render_template('dialog.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role_id = request.form.get('role_id', 0, type=int)

        new_user = User(username=username, email=email, password=password, role_id=role_id)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.password == password:
            session['user_id'] = user.id
            return redirect(url_for('home'))
        return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))


@app.route('/admin/users')
@admin_required
def admin_users():
    users = User.query.all()
    roles = UserRole.query.all()
    return render_template('admin_users.html', users=users, roles=roles)


@app.route('/admin/add_user', methods=['GET', 'POST'])
@admin_required
def admin_add_user():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role_id = request.form.get('role_id', 1, type=int)

        new_user = User(username=username, email=email, password=password, role_id=role_id)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('admin_users'))

    roles = UserRole.query.all()
    return render_template('admin_add_user.html', roles=roles)


@app.route('/admin/edit_user/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def admin_edit_user(user_id):
    user = User.query.get_or_404(user_id)

    if request.method == 'POST':
        user.username = request.form['username']
        user.email = request.form['email']
        user.role_id = request.form.get('role_id', user.role_id, type=int)

        if request.form['password']:
            user.password = request.form['password']

        db.session.commit()
        return redirect(url_for('admin_users'))


@app.route('/admin/delete_user/<int:user_id>')
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)

    if user.id != session['user_id']:
        db.session.delete(user)
        db.session.commit()

    return redirect(url_for('admin_users'))


@app.route('/statistics')
def statistics():
    return render_template('statistics.html')


def init_roles():
    if UserRole.query.count() == 0:
        roles = ['admin', 'user']
        for role_name in roles:
            role = UserRole(role_name=role_name)
            db.session.add(role)
        db.session.commit()


def create_admin():
    admin_role = UserRole.query.filter_by(role_name='admin').first()
    if admin_role and User.query.filter_by(role_id=admin_role.id).count() == 0:
        admin = User(
            username='admin',
            email='admin@example.com',
            password='admin123',
            role_id=admin_role.id
        )
        db.session.add(admin)
        db.session.commit()


if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    app.run(debug=True)