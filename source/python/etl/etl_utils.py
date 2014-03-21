__author__ = 'franpena'


class ETLUtils:
    def __init__(self):
        pass

    @staticmethod
    def drop_fields(fields, dictionary_list):
        """
        Removes the specified fields from every dictionary in the list records

        :rtype : void
        :param fields: a list of strings, which contains the fields that are
        going to be removed from every dictionary in the list records
        :param dictionary_list: a list of dictionaries
        """
        for record in dictionary_list:
            for field in fields:
                del (record[field])