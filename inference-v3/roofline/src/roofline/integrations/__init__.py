"""Registered execution integrations, imported only inside an executor."""


def integration(name):
    if name == "magnitude":
        from .magnitude import Magnitude

        return Magnitude
    if name == "llama.cpp":
        from .llama import Llama

        return Llama
    raise ValueError(f"unknown engine integration: {name}")
