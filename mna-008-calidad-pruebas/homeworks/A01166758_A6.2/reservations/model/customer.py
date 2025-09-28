from pydantic import BaseModel
from .sqlite3_adapter import execute, select


class Customer(BaseModel):
    id: int
    first_name: str
    last_name: str

    def save(self) -> int:
        sql = """INSERT INTO customer (
            first_name,
            last_name)
            VALUES(?, ?)"""

        execute(
            sql,
            (
                self.first_name,
                self.last_name,
            ),
        )

        last = "SELECT max(id) FROM customer"
        for id in select(last):
            self.id = id[0]
            return id

    def update(self, id: int):
        sql = """UPDATE customer SET
                first_name = ?,
                last_name = ? WHERE id = ?"""

        execute(
            sql,
            (
                self.first_name,
                self.last_name,
                id,
            ),
        )

        last = "SELECT max(id) FROM customer"
        for id in select(last):
            return id

    @staticmethod
    def delete(id: int):
        sql = """DELETE FROM customer WHERE id = ?"""
        execute(sql, (id,))

    @staticmethod
    def show(id: int):
        sql = """SELECT * from customer WHERE id = ?"""
        for it in select(sql, (id,)):
            h = Customer(
                id=it[0],
                first_name=it[1],
                last_name=it[2]
            )
            return h

    @staticmethod
    def showAll():
        sql = """SELECT * from customer"""
        result = []
        for it in select(sql):
            h = Customer(
                id=it[0],
                first_name=it[1],
                last_name=it[2]
            )
            result.append((it[0], h))
        return result
