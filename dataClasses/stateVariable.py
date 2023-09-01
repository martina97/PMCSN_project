class StateVariables:   # rappresenta il singolo blocco / centro
    def __init__(self, number_in_center, server_state):
        self.number_in_center = number_in_center  # popolazione totale nel centro
        self.server_state = server_state  # stato di ogni servente: idle (0) o busy (1), è un array di server

    def __str__(self):
        return f"StateVariables: number_in_center ={self.number_in_center}, server_state={self.server_state}"
