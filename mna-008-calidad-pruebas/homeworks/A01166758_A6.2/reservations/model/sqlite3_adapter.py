import sqlite3


DB = "reservations.db"


def start_db():
    with sqlite3.connect(DB) as conn:
        try:
            conn.execute(
                """CREATE TABLE hotels (
                    id integer not null primary key autoincrement,
                    name varchar(50),
                    description varchar(50),
                    max_floor integer,
                    max_rooms integer,
                    max_occupation integer
                );"""
            )

            conn.execute(
                """CREATE TABLE customer (
                    id integer not null primary key autoincrement,
                    first_name varchar(50),
                    last_name varchar(50)
                );"""
            )

            conn.execute(
                """CREATE TABLE reservations (
                    id integer not null primary key autoincrement,
                    customer_id integer not null,
                    hotel_id integer not null,
                    since DATETIME not null,
                    until DATETIME not null,
                    canceled BOOLEAN
                );"""
            )

            conn.commit()
        except sqlite3.OperationalError as e:
            print(e)
            pass


def execute(sql, *args):
    with sqlite3.connect(DB) as conn:
        try:
            conn.execute(sql, *args)
            conn.commit()
        except sqlite3.OperationalError as e:
            raise Exception(e)


def select(sql, *args):
    with sqlite3.connect(DB) as conn:
        try:
            if args:
                results = conn.execute(sql, *args)
            else:
                results = conn.execute(sql)
            for result in results:
                yield result
        except sqlite3.OperationalError as e:
            raise Exception(e)
