from cstr_experiment import run_closed_loop_simulation

if __name__ == "__main__":
    run_closed_loop_simulation(rrlb=False)
    run_closed_loop_simulation(rrlb=True)
