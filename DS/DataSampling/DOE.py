import lhs
lhs(4, criterion='center')
lhs(4, samples=10, criterion='center')

def ascending(order_square, top_left_number):
    """If order_square > top_left_number"""
    rows_counter = 0
    for rows in range(1, order_square + 1):
        latin_number = top_left_number + (rows - 1)
        if latin_number > order_square:
            latin_number = 1 + rows_counter
            rows_counter += 1
        row_counter = 0
        for row in range(1, order_square + 1):
            print(latin_number,)
            latin_number += 1
            if latin_number > order_square:
                latin_number = 1 + row_counter
                row_counter += 1
        print


ascending(order_square, top_left_number)

