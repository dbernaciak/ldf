forecasts_lik = [
    -2.302585093 -4.605170186 -4.605170186 -1.609437912 -4.605170186 -1.771956842 -1.609437912 -1.386294361 -1.386294361 -2.302585093 -1.386294361;
    -0.916290732 -1.203972804 -1.609437912 -2.995732274 -4.605170186 -2.302585093 -4.828313737 -1.386294361 -1.609437912 -2.991548932 -0.801038303;
    -2.302585093 -3.688879454 -3.688879454 -1.897119985 -2.9 -2.302585093 -4.1 -1.1 -0.1 -1.4 -1.897119985;
    -4.605170186 -4.605170186 -4.605170186 -4.605170186 -4.605170186 -0.9 -4.605170186 -0.510825624 -0.287682072, -1.491654877, -1.897119985;
    -1.203972804 -1.203972804 -1.290984181 -2.302585093 -4.605170186 -1 -4.605170186 -1.049822124 -2.302585093 -2.995732274 -0.597837001];

discount_factors = [1, 0.5];
activation_functions = ["softmax"];
[x, ~, ~] = ldf(forecasts_lik, 1, discount_factors, activation_functions);
assert(max(abs(sum(x.') - [-24.54868304 -21.48288988])) < 10^-8)

activation_functions = ["argmax"];
[x, ~, ~] = ldf(forecasts_lik, 1, discount_factors, activation_functions);
assert(max(abs(sum(x.') - [-24.57489777 -25.00121507])) < 10^-8) 

activation_functions = ["softmax", "softmax"];
[x, ~, ~] = ldf(forecasts_lik, 2, discount_factors, activation_functions);
assert(max(abs(sum(x.') - [-22.13047414 -22.23379625])) < 10^-8) 

activation_functions = ["argmax", "argmax"];
[x, ~, ~] = ldf(forecasts_lik, 2, discount_factors, activation_functions);
assert(max(abs(sum(x.') - [-25.87418075 -24.27844848])) < 10^-8) 

activation_functions = ["softmax", "softmax", "softmax"];
[x, ~, ~] = ldf(forecasts_lik, 3, discount_factors, activation_functions);
assert(max(abs(sum(x.') - [-22.180801356798145, -22.17890856487699])) < 10^-8) 