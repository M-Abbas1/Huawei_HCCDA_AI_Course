def print_details(**student_data):
    """
        Input: studnet data using keyword argument
        output: print student detail
        return = None
    """
    for key, value in student_data.items():
        print(f"{key:10s}     {value}")


print_details()

print()
