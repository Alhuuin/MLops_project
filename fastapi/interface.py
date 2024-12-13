import tkinter as tk
from tkinter import ttk, messagebox
import requests
from chatting.chatting import (
    Agent
)
from decouple import config

API_BASE_URL = "http://127.0.0.1:8000"
token = config("VALID_TOKEN")

class FunctionInterface:
    def __init__(self):
        self.agents = []
        
        self.dialog = None
        self.root = tk.Tk()
        self.root.title("Chatting interface")
        self.root.geometry("1920x1080")

        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        #=====[ Agent creation ]=====#

        # Agent Creation Frame
        self.agent_frame = ttk.LabelFrame(self.main_frame, text="Create New Agent", padding="5")
        self.agent_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)

        # Agent Name
        ttk.Label(self.agent_frame, text="Name:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        self.name_entry = ttk.Entry(self.agent_frame, width=30)
        self.name_entry.grid(row=0, column=1, padx=5, pady=2)

        # Agent Personality
        ttk.Label(self.agent_frame, text="Personality:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        self.personality_text = tk.Text(self.agent_frame, height=3, width=30)
        self.personality_text.grid(row=1, column=1, padx=5, pady=2)

        # Agent Gender
        ttk.Label(self.agent_frame, text="Gender:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        self.gender_var = tk.StringVar()
        self.gender_combo = ttk.Combobox(self.agent_frame, textvariable=self.gender_var, values=["male", "female"], state="readonly", width=27)
        self.gender_combo.grid(row=2, column=1, padx=5, pady=2)

        # Create Agent Button
        self.create_agent_btn = ttk.Button(
            self.agent_frame,
            text="Create Agent",
            command=lambda: [
                self.agents.append(Agent(
                    name=self.name_entry.get(),
                    agent_id=f"agent_{self.agent_counter:03d}",
                    user_input=self.personality_text.get("1.0", tk.END).strip(),
                    gender=self.gender_var.get()
                )),
                update_agent_list(),
                update_chat_agents()
            ]
        )
        self.create_agent_btn.grid(row=3, column=0, columnspan=2, pady=5)

        # Create Test Agents Button
        self.test_agents_btn = ttk.Button(
            self.agent_frame,
            text="Create Test Agents", 
            command=lambda: [
                self.agents.extend([
                    Agent("Sami", f"agent_{self.agent_counter:03d}", 
                          "Sami is a man who likes horses and love to go hiking. He also loves fishing. He has an opinion on everything and is kind most of the time.", "male"),
                    Agent("Eliana", f"agent_{self.agent_counter+1:03d}",
                          "Eliana is a woman who loves to eat and sleep. She is very VERY knowledgeable about ducks. She is the sweetest person on earth and would never lie or attack someone.", "female"),
                    Agent("Alice", f"agent_{self.agent_counter+2:03d}",
                          "Alice is a woman who loves cats and crochet. She doesn't like cooking at all. She is nice to people unless they are unkind to her first.", "female"),
                    Agent("Juliette", f"agent_{self.agent_counter+3:03d}",
                          "Juliette is a woman who loves to travel and learn new things. She is very open minded and always willing to help. She has been to every country in the world.", "female")
                ]),
                self.agents[-4].opinions.update({
                    "Eliana": "He likes Eliana and thinks her cooking advices are great",
                    "Alice": "He enjoys having conversations with her about his hobbies, he likes to fight with her about it",
                    "Juliette": "He likes that she's always willing to help"
                }),
                self.agents[-3].opinions.update({
                    "Sami": "She thinks Sami is cool and she likes his cooking",
                    "Alice": "She's happy to have found someone who likes animals as much as she does",
                    "Juliette": "She likes that she's always willing to help"
                }),
                self.agents[-2].opinions.update({
                    "Sami": "She likes hearing him talk about his hiking adventures",
                    "Eliana": "She likes that she's a kind and calm person",
                    "Juliette": "She likes that she's always willing to help and loves to talk about her travels"
                }),
                self.agents[-1].opinions.update({
                    "Sami": "She likes that he's always willing to help",
                    "Eliana": "She likes that she's a kind and calm person",
                    "Alice": "She likes that she's always willing to help"
                }),
                setattr(self, 'agent_counter', self.agent_counter + 3),
                update_agent_list(),
                update_chat_agents()
            ]
        )
        self.test_agents_btn.grid(row=4, column=0, columnspan=2, pady=5)

        # Counter for agent IDs
        self.agent_counter = 1
        
        #=====[Agent list]=====#
        
        # Agent List Frame
        self.agent_list_frame = ttk.LabelFrame(self.main_frame, text="Current Agents", padding="5")
        self.agent_list_frame.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky=tk.N+tk.S)

        # Agent List
        self.agent_listbox = tk.Listbox(self.agent_list_frame, width=40, height=15)
        self.agent_listbox.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Agent Details Text
        self.agent_details = tk.Text(self.agent_list_frame, width=40, height=5)
        self.agent_details.pack(padx=5, pady=5, fill=tk.BOTH)

        # Opinions Frame
        self.opinions_frame = ttk.LabelFrame(self.agent_list_frame, text="Opinions", padding="5")
        self.opinions_frame.pack(padx=5, pady=5, fill=tk.BOTH)

        # Update agent list function
        def update_agent_list():
            self.agent_listbox.delete(0, tk.END)
            for agent in self.agents:
                self.agent_listbox.insert(tk.END, f"{agent.name} ({agent.agent_id}) - {agent.gender}")

        # Function to display agent details and opinions
        def show_agent_details(event):
            selection = self.agent_listbox.curselection()
            if selection:
                index = selection[0]
                selected_agent = self.agents[index]
                
                # Update details
                self.agent_details.delete(1.0, tk.END)
                self.agent_details.insert(tk.END, f"Personality:\n{selected_agent.user_input}")
                
                # Clear and recreate opinion widgets
                for widget in self.opinions_frame.winfo_children():
                    widget.destroy()
                
                # Create opinion entries for other agents
                row = 0
                for agent in self.agents:
                    if agent != selected_agent:
                        ttk.Label(self.opinions_frame, text=f"Opinion on {agent.name}:").grid(row=row, column=0, padx=5, pady=2, sticky=tk.W)
                        opinion_entry = ttk.Entry(self.opinions_frame, width=30)
                        opinion_entry.grid(row=row, column=1, padx=5, pady=2)
                        
                        # Pre-fill existing opinion if it exists
                        if agent.name in selected_agent.opinions:
                            opinion_entry.insert(0, selected_agent.opinions[agent.name])
                            
                        # Save opinion on entry change
                        def save_opinion(agent_name=agent.name, entry=opinion_entry):
                            selected_agent.opinions[agent_name] = entry.get()
                            
                        opinion_entry.bind('<KeyRelease>', lambda e, save=save_opinion: save())
                        row += 1

        # Bind listbox selection to show details
        self.agent_listbox.bind('<<ListboxSelect>>', show_agent_details)

        # Bind update to create agent button
        self.create_agent_btn.config(command=lambda: [
            self.agents.append(Agent(
                name=self.name_entry.get(),
                agent_id=f"agent_{self.agent_counter:03d}", 
                user_input=self.personality_text.get("1.0", tk.END).strip(),
                gender=self.gender_var.get()
            )),
            update_agent_list(),
            setattr(self, 'agent_counter', self.agent_counter + 1)
        ])
        
        #=====[Start Chat]=====#
        
        # Create chat frame
        chat_frame = ttk.LabelFrame(self.root, text="Create Chat", padding="10")
        chat_frame.grid(row=0, column=3, rowspan=3, padx=10, pady=5, sticky="nsew")

        # Agent selection (multiple)
        ttk.Label(chat_frame, text="Select Agents:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.chat_agents_listbox = tk.Listbox(chat_frame, selectmode=tk.MULTIPLE, height=5)
        self.chat_agents_listbox.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)

        def update_chat_agents():
            self.chat_agents_listbox.delete(0, tk.END)
            for agent in self.agents:
                self.chat_agents_listbox.insert(tk.END, agent.name)
        
        # Bind update_chat_agents to agent list changes
        # Create a separate binding for chat agents update to avoid interfering with show_agent_details
        self.agent_listbox.bind('<<ListboxSelect>>', lambda e: [show_agent_details(e), update_chat_agents()])
        update_chat_agents()

        # Subject entry
        ttk.Label(chat_frame, text="Subject:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.subject_entry = ttk.Entry(chat_frame, width=50)
        self.subject_entry.grid(row=2, column=1, sticky="ew", pady=5)

        # Location entry
        ttk.Label(chat_frame, text="Location (optional):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.location_entry = ttk.Entry(chat_frame, width=50)
        self.location_entry.grid(row=3, column=1, sticky="ew", pady=5)

        # Memory checkbox
        self.use_memory_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(chat_frame, text="Use Memory", variable=self.use_memory_var).grid(row=4, column=0, columnspan=2, pady=5)

        # Chat output
        ttk.Label(chat_frame, text="Chat Output:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.chat_output = tk.Text(chat_frame, height=10, width=50, wrap=tk.WORD)
        self.chat_output.grid(row=6, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Add scrollbar for chat output
        chat_scrollbar = ttk.Scrollbar(chat_frame, orient="vertical", command=self.chat_output.yview)
        chat_scrollbar.grid(row=6, column=2, sticky="ns")
        self.chat_output.configure(yscrollcommand=chat_scrollbar.set)

        # Start chat button
        def start_chat():
            selected_indices = self.chat_agents_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "Please select at least one agent")
                return
            
            selected_agents = [self.agents[i] for i in selected_indices]
            subject = self.subject_entry.get()
            location = self.location_entry.get() or None
            use_memory = self.use_memory_var.get()

            # Clear previous output
            self.chat_output.delete(1.0, tk.END)
            
            # Start chat and display result
            # Show loading message
            self.chat_output.insert(tk.END, "Generating chat, please wait...")
            self.chat_output.update()
            
            # Run chat in a separate thread to avoid freezing UI
            def run_chat():
                try:
                    payload = {
                        "agents": [a.to_dict() for a in selected_agents],
                        "subject": subject,
                        "location": location,
                        "use_memory": use_memory,
                    }

                    headers = {
                        "Authorization": token,
                        "Content-Type": "application/json"
                    }

                    response = requests.post(f"{API_BASE_URL}/chat", json=payload, headers= headers)
                    response.raise_for_status()
                    
                    result = response.json()
                    dialog = result.get("dialog", "No response generated")

                    updated_agents = result.get("updated_agents", [])
                    for agent_data in updated_agents:
                        for agent in selected_agents:
                            if agent.agent_id == agent_data["agent_id"]:
                                agent.opinions.update(agent_data.get("opinions", {}))

                except requests.exceptions.RequestException as e:
                    dialog = f"Error: {e}"

                self.chat_output.delete(1.0, tk.END)
                self.chat_output.insert(tk.END, dialog if dialog else "No response generated")
            
            import threading
            chat_thread = threading.Thread(target=run_chat)
            chat_thread.start()
                        
        ttk.Button(chat_frame, text="Start Chat", command=start_chat).grid(row=7, column=0, columnspan=2, pady=10)

    def run(self):
        """Start the interface"""
        self.root.mainloop()

if __name__ == "__main__":
    # Example usage
    interface = FunctionInterface()
    interface.run()