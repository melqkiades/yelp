__author__ = 'fpena'


class Vote:

    def __init__(self):
        self._user = None
        self._item = None
        self._rating = None
        self._date = None
        self._word_list = []

    @property
    def user(self):
        """
        :rtype: int
        """
        return self._user

    @user.setter
    def user(self, value):
        """
        :type value: int
        """
        self._user = value

    @property
    def item(self):
        """
        :rtype: int
        """
        return self._item

    @item.setter
    def item(self, value):
        """
        :type value: int
        """
        self._item = value

    @property
    def rating(self):
        """
        :rtype: float
        """
        return self._rating

    @rating.setter
    def rating(self, value):
        """
        :type value: float
        """
        self._rating = value

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, value):
        self._date = value

    @property
    def word_list(self):
        """
        :rtype: list[str]
        """
        return self._word_list

    @word_list.setter
    def word_list(self, value):
        """
        :type value: list[str]
        """
        self._word_list = value