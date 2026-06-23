# circular_economy_RL_lib/simulator.py

import numpy as np
from config import config, init_historical_data
from surrogate_models import SurrogateModel

class Manufacturing_Simulator:
    def __init__(self):
        for key, value in config.items():
            setattr(self, key, value)
        self.surrogate_model = SurrogateModel()
        
        # Leader physical mechanism state
        self.market_mechanism = {
            "spot_mult": 1.0,
            "uc_mult": 1.0,
            "tx_mult": 1.0,
            "waste_penalty": 0.0
        }
        
    def reset(self):
        self.t = self.history_length
        data_length = self.history_length + self.episode_length + 1
        
        general_shape = (self.num_commodities, data_length)
        individual_shape = (self.num_agents, self.num_commodities, data_length)
        pair_shape = (self.num_agents, self.num_agents, self.num_commodities, data_length)

        # Retrieve base historical spot data
        historical_data = init_historical_data()
        base_spot = np.repeat(historical_data['spot_price'], repeats=data_length, axis=1)
        
        # Apply Market Authority multi-pliers to adjust dynamic costs
        self.spot_price = base_spot * self.market_mechanism["spot_mult"]
        self.uc_p = self.UC * self.market_mechanism["uc_mult"] * self.spot_price
        self.tx_p = self.TX_P * self.market_mechanism["tx_mult"] * self.spot_price

        self.price = np.zeros(shape=individual_shape)
        self.waste_price = np.zeros(shape=individual_shape)
        self.q = np.zeros(shape=pair_shape)
        self.waste_q = np.zeros(shape=pair_shape)
        self.spot_q = np.zeros(shape=individual_shape)
        self.actual_d = np.zeros(shape=pair_shape)
        self.waste_actual_d = np.zeros(shape=pair_shape)

        self.inv = np.zeros(shape=individual_shape)
        self.waste_inv = np.zeros(shape=individual_shape)
        self.inv_buy = np.zeros(shape=individual_shape)
        self.waste_inv_buy = np.zeros(shape=individual_shape)

        self.eco_u = np.zeros(shape=individual_shape)
        self.tx_u = np.zeros(shape=individual_shape)
        self.wastewater = np.zeros(shape=(1, 1, data_length))

        self.inv[:, :, self.t] = self.inv[:, :, self.t] + self.INIT_INV
        self.waste_inv = self.inv.copy()

        return self.get_seller_state()

    def set_market_mechanism(self, mechanism):
        if mechanism is not None:
            self.market_mechanism = mechanism

    def get_seller_state(self):
        seller_states = []
        start_time = self.t - self.history_length
        for n in range(self.num_agents):
            p = self.spot_price[..., start_time:self.t].flatten()
            e = self.price[..., start_time:self.t].flatten()
            ew = self.waste_price[..., start_time:self.t].flatten()
            q = self.q[:, n, :, start_time:self.t].flatten()
            qw = self.waste_q[:, n, :, start_time:self.t].flatten()
            q_ = self.q[n, :, :, start_time:self.t].flatten()
            qw_ = self.waste_q[n, :, :, start_time:self.t].flatten()
            qs = self.spot_q[n, :, start_time:self.t].flatten()
            d = self.actual_d[n, :, :, start_time:self.t].flatten()
            dw = self.waste_actual_d[n, :, :, start_time:self.t].flatten()
            I = self.inv[n, :, start_time:self.t+1].flatten()
            Iw = self.waste_inv[n, :, start_time:self.t+1].flatten()
            u_eco = self.eco_u[n, :, start_time:self.t].flatten()
            u_tx = self.tx_u[n, :, start_time:self.t].flatten()   
            
            state_flat = np.concatenate((p, e, ew, q, qw, q_, qw_, qs, d, dw, I, Iw, u_eco, u_tx))
            seller_states.append(state_flat)
        return np.array(seller_states)

    def action_conversion(self, keys, actions):
        conv_actions = {k: np.zeros((self.num_agents, length), dtype=actions.dtype) for k, length in keys.items()}
        for i in range(self.num_agents):
            start = 0
            for key, length in keys.items():
                conv_actions[key][i] = actions[i, start:start + length]
                start += length
        return conv_actions

    def step_sell(self, seller_states, orig_seller_actions):
        keys = ['price', 'waste_price']
        key_len_dict = {k: self.num_commodities for k in keys}
        seller_actions = self.action_conversion(key_len_dict, orig_seller_actions)
        
        # Apply Market Authority parameters directly to state
        self.price[..., self.t] = seller_actions["price"]
        # Apply the leader waste penalty on waste prices to scale dynamic transactions
        self.waste_price[..., self.t] = np.maximum(
            seller_actions["waste_price"] - self.market_mechanism["waste_penalty"], 0.0
        )
        return self.get_buyer_state(keys, seller_states, seller_actions)

    def get_buyer_state(self, keys, seller_states, seller_actions):
        buyer_states = []
        for n in range(self.num_agents):
            state_flat = seller_states[n]
            state_flat = np.concatenate((state_flat, self.spot_price[:, self.t]))
            for key in keys:
                state_flat = np.concatenate((state_flat, seller_actions[key].flatten()))
            buyer_states.append(state_flat)
        return np.array(buyer_states)

    def step_buy(self, buyer_states, orig_buyer_actions):
        keys = ['q', 'waste_q', 'spot_q']
        nc = (self.num_agents - 1) * self.num_commodities
        lengths = [nc, nc, self.num_commodities]
        key_len_dict = {k: v for k, v in zip(keys, lengths)}
        buyer_actions = self.action_conversion(key_len_dict, orig_buyer_actions)
        for k, arr in buyer_actions.items():
            if k == 'spot_q':
                continue
            new_actions = np.zeros((self.num_agents, self.num_agents, self.num_commodities))
            arr = arr.reshape(self.num_agents, self.num_agents - 1, self.num_commodities)
            for i in range(self.num_agents):
                i_list = list(range(self.num_agents))
                i_list.remove(i)
                new_actions[i, i_list] = arr[i]
            buyer_actions[k] = new_actions

        for key, value in buyer_actions.items():
            getattr(self, key)[..., self.t] = value

        trans_states = self.get_trans_state(keys, buyer_states, buyer_actions)
        buyer_rewards = self.get_buyer_reward()
        seller_rewards = self.get_seller_reward()
        return trans_states, buyer_rewards, seller_rewards

    def get_seller_reward(self):
        actual_d = self.actual_d[:, :, :, self.t].sum(axis=0)
        waste_actual_d = self.waste_actual_d[:, :, :, self.t].sum(axis=0)
        reward = (self.price[:, :, self.t] * actual_d).sum(axis=1)
        reward += (self.waste_price[:, :, self.t] * waste_actual_d).sum(axis=1)
        return reward * self.RWD_SCALE

    def get_buyer_reward(self):
        e_reshape = self.price[:, :, self.t].reshape(1, self.num_agents, self.num_commodities)
        ew_reshape = self.waste_price[:, :, self.t].reshape(1, self.num_agents, self.num_commodities)
        p_reshape = self.spot_price[:, self.t].reshape(1, self.num_commodities)

        reward = -np.sum(self.actual_d[:, :, :, self.t] * e_reshape, axis=(1, 2))
        reward -= np.sum(self.waste_actual_d[:, :, :, self.t] * ew_reshape, axis=(1, 2))
        reward -= np.sum(self.spot_q[:, :, self.t] * p_reshape, axis=1)
        reward += self.LAMBDA * np.sum(self.actual_d[:, :, :, self.t] - self.q[:, :, :, self.t], axis=(1, 2))
        reward += self.LAMBDA * np.sum(self.waste_actual_d[:, :, :, self.t] - self.waste_q[:, :, :, self.t], axis=(1, 2))
        return reward * self.RWD_SCALE

    def get_trans_state(self, keys, buyer_states, buyer_actions):
        actual_d = self.calc_actual_sold(self.q[:, :, :, self.t], self.inv[:, :, self.t])
        actual_dw = self.calc_actual_sold(self.waste_q[:, :, :, self.t], self.waste_inv[:, :, self.t])
        inv_buy = self.calc_inv_buy(self.inv[:, :, self.t], actual_d)
        waste_inv_buy = self.calc_inv_buy(self.waste_inv[:, :, self.t], actual_dw)

        inv_buy = inv_buy + self.spot_q[:, :, self.t]
        trans_states = []
        for n in range(self.num_agents):
            state_flat = buyer_states[n]
            for key in keys:
                state_flat = np.concatenate([state_flat, buyer_actions[key][n].flatten()])
            state_flat = np.concatenate([state_flat, actual_d[n, :, :].flatten(), actual_dw[n, :, :].flatten()])
            state_flat = np.concatenate([state_flat, inv_buy[n, :].flatten(), waste_inv_buy[n, :].flatten()])
            trans_states.append(state_flat)
        
        self.actual_d[..., self.t] = actual_d
        self.waste_actual_d[..., self.t] = actual_dw
        self.inv_buy[..., self.t] = inv_buy
        self.waste_inv_buy[..., self.t] = waste_inv_buy
        return np.array(trans_states)

    def calc_actual_sold(self, q, I):
        d = np.zeros_like(q)
        for c in range(self.num_commodities):
            for n in range(self.num_agents):
                buys = np.array([q[m, n, c] for m in range(self.num_agents)])
                sorted_indices = np.argsort(-buys)
                cum_sum = 0
                for i in range(self.num_agents):
                    agent_i = sorted_indices[i]
                    if i == 0:
                        d[agent_i, n, c] = min(I[n, c], buys[agent_i])
                    else:
                        available_I = I[n, c] - cum_sum
                        if available_I <= 0:
                            break
                        d[agent_i, n, c] = min(available_I, buys[agent_i])
                    cum_sum += d[agent_i, n, c]
        return d

    def calc_inv_buy(self, I_bar, d):
        return I_bar + np.sum(d, axis=1) - np.sum(d, axis=0)

    def step_trans(self, trans_states, orig_trans_actions):
        keys = ['tx_u', 'eco_u']
        key_len_dict = {k: self.num_commodities for k in keys}
        trans_actions = self.action_conversion(key_len_dict, orig_trans_actions)
        trans_actions['tx_u'] = np.minimum(trans_actions['tx_u'], 0.5 * self.inv_buy[..., self.t])
        trans_actions['eco_u'] = np.minimum(trans_actions['eco_u'], 0.5 * self.inv_buy[..., self.t])
        
        for key, value in trans_actions.items():
            getattr(self, key)[..., self.t] = value

        u_bot, w_bot = self.apply_agent_surrogate(trans_actions['tx_u'])
        self.inv[:, :, self.t + 1] = np.maximum(
            self.inv_buy[:, :, self.t] - trans_actions['tx_u'] - trans_actions['eco_u'] + u_bot, 0.0
        )
        self.waste_inv[:, :, self.t + 1] = (1.0 - self.delta) * (self.waste_inv_buy[:, :, self.t] + w_bot)
        
        trans_rewards = self.get_trans_reward()
        self.t += 1
        seller_states = self.get_seller_state()
        
        done = False
        if self.t == self.episode_length:
            done = True
        return seller_states, trans_rewards, done

    def get_trans_reward(self):
        uc_p = self.uc_p[:, self.t].reshape(1, self.num_commodities)
        tx_p = self.tx_p[:, self.t].reshape(1, self.num_commodities)
        reward = np.sum(self.eco_u[:, :, self.t] * uc_p, axis=1)
        reward -= np.sum(self.tx_u[:, :, self.t] * tx_p, axis=1)
        return reward * self.RWD_SCALE

    def apply_agent_surrogate(self, tx_u):
        agent0 = tx_u[0]
        agent1 = tx_u[1]
        agent2 = tx_u[2]

        agents_final_output_vec = np.zeros_like(tx_u)
        agent0_surrogate_input_vec = np.array([agent0[5], agent0[7], agent0[2], agent0[0], agent0[1]])
        agent0_surrogate_output_vec = self.surrogate_model.get_apap_model_outputs(agent0_surrogate_input_vec.reshape(1,-1))
        agents_final_output_vec[0, [2, 6, 0]] = np.array(agent0_surrogate_output_vec)[0, [0, 1, 3]]
        self.wastewater[:, :, self.t] = np.array(agent0_surrogate_output_vec)[0, [3]]

        agent1_surrogate_input_vec = np.array([agent1[3], agent1[4], agent1[10], agent1[9]])
        agent1_surrogate_output_vec = self.surrogate_model.get_pap_model_outputs(agent1_surrogate_input_vec.reshape(1,-1))
        agents_final_output_vec[1, [11, 5]] = np.array(agent1_surrogate_output_vec)[0, [0, 1]]

        agent2_surrogate_input_vec = np.array([self.h2_surrogate_input_conversion(agent2[0], agent2[2])])
        agent2_surrogate_output_vec = self.surrogate_model.get_hyd_model_outputs(agent2_surrogate_input_vec.reshape(1,-1))
        agents_final_output_vec[2, [2, 8, 3, 0]] = np.array(agent2_surrogate_output_vec)[0, [0, 1, 3, 4]]

        agents_waste_final_output_vec = np.zeros_like(tx_u)
        return agents_final_output_vec, agents_waste_final_output_vec

    def h2_surrogate_input_conversion(self, water, acetic_acid):
        if water >= 19.0 * acetic_acid:
            return 20.0 * acetic_acid
        return 20.0 * water / 19.0

    def get_market_state(self):
        # Constructs the state observed by the MarketAuthority leader (aggregated over time t-1)
        t = self.t
        avg_price = self.price[:, :, max(0, t - 1)].mean(axis=0)
        avg_inv = self.inv[:, :, t].mean(axis=0)
        avg_waste_inv = self.waste_inv[:, :, t].mean(axis=0)
        avg_spot_price = self.spot_price[:, t]
        avg_tx = self.tx_u[:, :, max(0, t - 1)].mean(axis=0)
        return np.concatenate([avg_price, avg_inv, avg_waste_inv, avg_spot_price, avg_tx])

    def get_upper_level_reward(self):
        # Represents the societal targets compiled over the step
        profit = np.sum(self.price[:, :, self.t] * self.actual_d[:, :, :, self.t])
        wastewater = np.sum(self.wastewater[:, :, self.t])
        waste_inventory = np.sum(self.waste_inv[:, :, self.t])
        imbalance = np.var(np.sum(self.inv[:, :, self.t], axis=1))
        return {
            "profit": profit,
            "wastewater": wastewater,
            "waste_inventory": waste_inventory,
            "imbalance": imbalance
        }
