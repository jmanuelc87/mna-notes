from pydantic import BaseModel
from .customer import Customer
from .hotel import Hotel
from datetime import datetime
from .sqlite3_adapter import execute, select


class Reservation(BaseModel):
    id: int
    customer: Customer
    hotel: Hotel
    since: datetime
    to: datetime
    canceled: bool

    def create_reservation(self):
        sql = """INSERT INTO reservations (
            customer_id,
            hotel_id,
            since,
            until) VALUES (?, ?, ?, ?)"""

        execute(
            sql,
            (
                self.customer.id,
                self.hotel.id,
                str(self.since),
                str(self.to),
            ),
        )

        last = "SELECT max(id) FROM reservations"
        for id in select(last):
            self.id = id[0]
            return id

    def cancel_reservation(self):
        sql = """UPDATE reservations
                SET canceled = true WHERE
                customer_id = ? and
                hotel_id = ? and
                id = ?"""

        execute(sql, (self.customer.id, self.hotel.id, self.id))

        self.canceled = True

        return True
