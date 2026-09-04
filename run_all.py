"""
Run every experiment in order.

  python run_all.py                 # all, lambda=1.0
  python run_all.py --lam=0.5
  python run_all.py --only=exp2
"""
import fire
import pytorch_lightning as pl
import config
import experiment_1, experiment_2, experiment_3, experiment_4, experiment_5, experiment_6


def main(lam=1.0, max_size=6, max_epochs=None, screening_epochs=None,
         seed=config.SEED, only=None):
    pl.seed_everything(seed, workers=True)
    print(f"seed={seed}  max_epochs={max_epochs or config.MAX_EPOCHS}  "
          f"screening_epochs={screening_epochs or config.SCREENING_EPOCHS}  lambda={lam}")

    steps = {
        "exp1": lambda: experiment_1.run(max_epochs, seed),
        "exp2": lambda: experiment_2.run(max_epochs, seed),
        "exp3": lambda: experiment_3.run(max_size, max_epochs, screening_epochs, seed),
        "exp4": lambda: experiment_4.run(None, max_epochs, seed),
        "exp5": lambda: experiment_5.run(max_epochs, screening_epochs, seed),
        "exp6a": lambda: experiment_6.run("6a", lam, max_size, max_epochs,
                                          screening_epochs, seed),
        "exp6b": lambda: experiment_6.run("6b", lam, max_size, max_epochs,
                                          screening_epochs, seed),
    }
    order = [only] if only else ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6a", "exp6b"]
    for k in order:
        steps[k]()
    print(f"\nDone. CSVs in {config.RESULTS_DIR}/")


if __name__ == "__main__":
    fire.Fire(main)
