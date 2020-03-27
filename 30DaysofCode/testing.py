# -*- coding: utf-8 -*-

from random import randint


class TestDataEmptyArray(object):
    @staticmethod
    def get_array():
        return []


class TestDataUniqueValues(object):
    _data = set()
    while len(_data) < 10:
        _data.add(randint(0, 100))

    @staticmethod
    def get_array():
        data = TestDataUniqueValues._data
        return list(data)

    @staticmethod
    def get_expected_result():
        data = TestDataUniqueValues.get_array()
        return data.index(min(data))


class TestDataExactlyTwoDifferentMinimums(object):

    _data = set()
    while len(_data) < 10:
        _data.add(randint(0, 100))

    _new_data = list(_data)
    _new_data.append(min(_new_data))

    @staticmethod
    def get_array():
        data = TestDataExactlyTwoDifferentMinimums._new_data
        return data

    @staticmethod
    def get_expected_result():
        data = TestDataExactlyTwoDifferentMinimums.get_array()
        return data.index(min(data))
