from reservations.model.hotel import Hotel
from reservations.model.customer import Customer
from reservations.model.reservation import Reservation
from datetime import datetime, timedelta


def test_create_reservation():
    h = Hotel(
        id=0,
        name="Holiday Inn Express",
        description="",
        max_floor=10,
        max_rooms=100,
        max_occupation=1000
    )

    h.save()

    c = Customer(
        id=0,
        first_name="Juan Manuel",
        last_name="Carballo"
    )

    c.save()

    r = Reservation(
        id=0,
        customer=c,
        hotel=h,
        since=datetime.now() + timedelta(days=30),
        to=datetime.now() + timedelta(days=40),
        canceled=False
    )

    r.create_reservation()

    assert r.id != 0


def test_cancel_reservation():
    h = Hotel(
        id=0,
        name="Holiday Inn Express",
        description="",
        max_floor=10,
        max_rooms=100,
        max_occupation=1000
    )

    h.save()

    c = Customer(
        id=0,
        first_name="Juan Manuel",
        last_name="Carballo"
    )

    c.save()

    r = Reservation(
        id=0,
        customer=c,
        hotel=h,
        since=datetime.now() + timedelta(days=30),
        to=datetime.now() + timedelta(days=40),
        canceled=False
    )

    r.create_reservation()

    r.cancel_reservation()

    assert r.canceled is True
