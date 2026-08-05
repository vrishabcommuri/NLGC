
def pretty_print_elapsed(elapsed):
    days, rem = divmod(elapsed, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    if days:
        print(f"Execution time: {int(days)}d {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    elif hours:
        print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    elif minutes:
        print(f"Execution time: {int(minutes)}m {seconds:.2f}s")
    else:
        print(f"Execution time: {seconds:.2f}s")