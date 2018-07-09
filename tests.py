import unittest

from feature_extraction_bag_of_words import sym_kl_div

from utils.clean_data import clean_txt


class Test(unittest.TestCase):

    def test_txt_clean_apostrophe(self):
        t1 = "what's this?"
        t2 = "i've gotten it."
        t3 = "i can't move."
        t4 = "i don't cry."
        t5 = "i'm good."
        t6 = "i m good."
        t7 = "we're good."
        t8 = "we'd leave."
        t9 = "we'll leave."

        self.assertEqual(clean_txt(t1), 'what is this')
        self.assertEqual(clean_txt(t2), 'i have gotten it')
        self.assertEqual(clean_txt(t3), 'i cannot move')
        self.assertEqual(clean_txt(t4), 'i do not cry')
        self.assertEqual(clean_txt(t5), 'i am good')
        self.assertEqual(clean_txt(t6), 'i am good')
        self.assertEqual(clean_txt(t7), 'we are good')
        self.assertEqual(clean_txt(t8), 'we would leave')
        self.assertEqual(clean_txt(t9), 'we will leave')


    def test_sym_kl_div(self):
        x = [1, 1, 2]
        y = [2, 1, 2]
        sym_kl_div(x, y)


if __name__ == '__main__':
    unittest.main()